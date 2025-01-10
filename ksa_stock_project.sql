-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jan 10, 2025 at 01:30 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ksa_stock_project`
--

-- --------------------------------------------------------

--
-- Table structure for table `funds`
--

CREATE TABLE `funds` (
  `fund_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `available_balance` decimal(10,2) NOT NULL,
  `total_balance` decimal(10,2) NOT NULL,
  `last_updated` decimal(10,2) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `funds`
--

INSERT INTO `funds` (`fund_id`, `user_id`, `available_balance`, `total_balance`, `last_updated`) VALUES
(1, 1, 1000.00, 166000.00, 2024.00),
(2, 2, 815.00, 32123.00, 2025.00);

-- --------------------------------------------------------

--
-- Table structure for table `portfolio`
--

CREATE TABLE `portfolio` (
  `portfolio_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `stock_id` int(11) NOT NULL,
  `quantity` int(11) NOT NULL,
  `purchase_price` decimal(10,0) NOT NULL,
  `purchase_date` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `portfolio`
--

INSERT INTO `portfolio` (`portfolio_id`, `user_id`, `stock_id`, `quantity`, `purchase_price`, `purchase_date`) VALUES
(1, 1, 1, 100, 72, '2024-12-30 19:00:00'),
(2, 1, 2, 500, 28, '2024-12-30 19:00:00'),
(3, 1, 3, 1000, 8, '2024-12-30 19:00:00'),
(4, 1, 4, 50, 112, '2024-12-30 19:00:00'),
(5, 2, 2, 302, 28, '2025-01-05 12:22:09'),
(6, 2, 5, 201, 17, '2025-01-04 11:05:32'),
(7, 2, 6, 150, 26, '2024-12-30 19:00:00'),
(8, 2, 7, 75, 120, '2024-12-30 19:00:00'),
(11, 2, 4, 1, 112, '2025-01-03 19:00:00');

-- --------------------------------------------------------

--
-- Table structure for table `stock price history`
--

CREATE TABLE `stock price history` (
  `price_history_id` int(11) NOT NULL,
  `stock_id` int(11) NOT NULL,
  `price` decimal(10,2) NOT NULL,
  `date` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `stock price history`
--

INSERT INTO `stock price history` (`price_history_id`, `stock_id`, `price`, `date`) VALUES
(1, 1, 71.20, '2024-11-30 19:00:00'),
(2, 1, 71.15, '2024-12-01 19:00:00'),
(3, 1, 71.45, '2024-12-02 19:00:00'),
(4, 1, 72.10, '2024-12-03 19:00:00'),
(5, 1, 72.55, '2024-12-04 19:00:00'),
(6, 1, 72.90, '2024-12-05 19:00:00'),
(7, 1, 73.15, '2024-12-06 19:00:00'),
(8, 1, 73.26, '2024-12-07 19:00:00'),
(9, 1, 72.95, '2024-12-08 19:00:00'),
(10, 1, 72.40, '2024-12-09 19:00:00'),
(11, 1, 72.60, '2024-12-10 19:00:00'),
(12, 1, 72.55, '2024-12-11 19:00:00'),
(13, 1, 72.45, '2024-12-12 19:00:00'),
(14, 1, 72.30, '2024-12-13 19:00:00'),
(15, 1, 72.15, '2024-12-14 19:00:00'),
(16, 1, 71.94, '2024-12-15 19:00:00'),
(17, 1, 71.94, '2024-12-16 19:00:00'),
(18, 1, 71.94, '2024-12-17 19:00:00'),
(19, 1, 71.94, '2024-12-18 19:00:00'),
(20, 1, 71.94, '2024-12-19 19:00:00'),
(21, 1, 71.94, '2024-12-20 19:00:00'),
(22, 1, 71.50, '2024-12-21 19:00:00'),
(23, 1, 71.00, '2024-12-22 19:00:00'),
(24, 1, 70.62, '2024-12-23 19:00:00'),
(25, 1, 71.28, '2024-12-24 19:00:00'),
(26, 1, 70.95, '2024-12-25 19:00:00'),
(27, 1, 70.75, '2024-12-26 19:00:00'),
(28, 1, 70.40, '2024-12-27 19:00:00'),
(29, 1, 69.96, '2024-12-28 19:00:00'),
(30, 1, 71.28, '2024-12-29 19:00:00'),
(31, 2, 27.45, '2024-11-30 19:00:00'),
(32, 2, 27.65, '2024-12-01 19:00:00'),
(33, 2, 27.80, '2024-12-02 19:00:00'),
(34, 2, 27.90, '2024-12-03 19:00:00'),
(35, 2, 27.95, '2024-12-04 19:00:00'),
(36, 2, 28.00, '2024-12-05 19:00:00'),
(37, 2, 28.00, '2024-12-06 19:00:00'),
(38, 2, 28.00, '2024-12-07 19:00:00'),
(39, 2, 28.65, '2024-12-08 19:00:00'),
(40, 2, 28.45, '2024-12-09 19:00:00'),
(41, 2, 28.45, '2024-12-10 19:00:00'),
(42, 2, 28.45, '2024-12-11 19:00:00'),
(43, 2, 28.45, '2024-12-12 19:00:00'),
(44, 2, 28.45, '2024-12-13 19:00:00'),
(45, 2, 28.45, '2024-12-14 19:00:00'),
(46, 2, 28.20, '2024-12-15 19:00:00'),
(47, 2, 28.15, '2024-12-16 19:00:00'),
(48, 2, 28.50, '2024-12-17 19:00:00'),
(49, 2, 29.00, '2024-12-18 19:00:00'),
(50, 2, 28.60, '2024-12-19 19:00:00'),
(51, 2, 28.50, '2024-12-20 19:00:00'),
(52, 2, 28.40, '2024-12-21 19:00:00'),
(53, 2, 28.30, '2024-12-22 19:00:00'),
(54, 2, 28.20, '2024-12-23 19:00:00'),
(55, 2, 28.10, '2024-12-24 19:00:00'),
(56, 2, 28.00, '2024-12-25 19:00:00'),
(57, 2, 27.90, '2024-12-26 19:00:00'),
(58, 2, 27.85, '2024-12-27 19:00:00'),
(59, 2, 27.95, '2024-12-28 19:00:00'),
(60, 2, 28.05, '2024-12-29 19:00:00'),
(61, 3, 8.25, '2024-11-30 19:00:00'),
(62, 3, 8.20, '2024-12-01 19:00:00'),
(63, 3, 8.15, '2024-12-02 19:00:00'),
(64, 3, 8.10, '2024-12-03 19:00:00'),
(65, 3, 8.10, '2024-12-04 19:00:00'),
(66, 3, 8.10, '2024-12-05 19:00:00'),
(67, 3, 8.12, '2024-12-06 19:00:00'),
(68, 3, 8.15, '2024-12-07 19:00:00'),
(69, 3, 8.20, '2024-12-08 19:00:00'),
(70, 3, 8.35, '2024-12-09 19:00:00'),
(71, 3, 8.40, '2024-12-10 19:00:00'),
(72, 3, 8.30, '2024-12-11 19:00:00'),
(73, 3, 8.25, '2024-12-12 19:00:00'),
(74, 3, 8.20, '2024-12-13 19:00:00'),
(75, 3, 8.30, '2024-12-14 19:00:00'),
(76, 3, 8.25, '2024-12-15 19:00:00'),
(77, 3, 8.20, '2024-12-16 19:00:00'),
(78, 3, 8.18, '2024-12-17 19:00:00'),
(79, 3, 8.20, '2024-12-18 19:00:00'),
(80, 3, 8.17, '2024-12-19 19:00:00'),
(81, 3, 8.20, '2024-12-20 19:00:00'),
(82, 3, 8.18, '2024-12-21 19:00:00'),
(83, 3, 8.15, '2024-12-22 19:00:00'),
(84, 3, 8.17, '2024-12-23 19:00:00'),
(85, 3, 8.15, '2024-12-24 19:00:00'),
(86, 3, 8.17, '2024-12-25 19:00:00'),
(87, 3, 8.20, '2024-12-26 19:00:00'),
(88, 3, 8.22, '2024-12-27 19:00:00'),
(89, 3, 8.25, '2024-12-28 19:00:00'),
(90, 3, 8.26, '2024-12-29 19:00:00'),
(91, 4, 112.00, '2024-11-30 19:00:00'),
(92, 4, 112.20, '2024-12-01 19:00:00'),
(93, 4, 112.80, '2024-12-02 19:00:00'),
(94, 4, 113.50, '2024-12-03 19:00:00'),
(95, 4, 115.20, '2024-12-04 19:00:00'),
(96, 4, 114.80, '2024-12-05 19:00:00'),
(97, 4, 115.00, '2024-12-06 19:00:00'),
(98, 4, 118.80, '2024-12-07 19:00:00'),
(99, 4, 118.20, '2024-12-08 19:00:00'),
(100, 4, 118.80, '2024-12-09 19:00:00'),
(101, 4, 118.40, '2024-12-10 19:00:00'),
(102, 4, 117.80, '2024-12-11 19:00:00'),
(103, 4, 116.00, '2024-12-12 19:00:00'),
(104, 4, 115.40, '2024-12-13 19:00:00'),
(105, 4, 115.60, '2024-12-14 19:00:00'),
(106, 4, 115.80, '2024-12-15 19:00:00'),
(107, 4, 115.60, '2024-12-16 19:00:00'),
(108, 4, 115.80, '2024-12-17 19:00:00'),
(109, 4, 113.20, '2024-12-18 19:00:00'),
(110, 4, 112.40, '2024-12-19 19:00:00'),
(111, 4, 111.20, '2024-12-20 19:00:00'),
(112, 4, 112.00, '2024-12-21 19:00:00'),
(113, 4, 112.40, '2024-12-22 19:00:00'),
(114, 4, 112.00, '2024-12-23 19:00:00'),
(115, 4, 111.80, '2024-12-24 19:00:00'),
(116, 4, 111.20, '2024-12-25 19:00:00'),
(117, 4, 110.80, '2024-12-26 19:00:00'),
(118, 4, 111.60, '2024-12-27 19:00:00'),
(119, 4, 111.80, '2024-12-28 19:00:00'),
(120, 4, 111.60, '2024-12-29 19:00:00'),
(121, 5, 17.80, '2024-11-30 19:00:00'),
(122, 5, 17.60, '2024-12-01 19:00:00'),
(123, 5, 17.85, '2024-12-02 19:00:00'),
(124, 5, 18.00, '2024-12-03 19:00:00'),
(125, 5, 18.00, '2024-12-04 19:00:00'),
(126, 5, 18.05, '2024-12-05 19:00:00'),
(127, 5, 18.10, '2024-12-06 19:00:00'),
(128, 5, 18.14, '2024-12-07 19:00:00'),
(129, 5, 18.20, '2024-12-08 19:00:00'),
(130, 5, 18.00, '2024-12-09 19:00:00'),
(131, 5, 17.85, '2024-12-10 19:00:00'),
(132, 5, 17.80, '2024-12-11 19:00:00'),
(133, 5, 17.75, '2024-12-12 19:00:00'),
(134, 5, 17.70, '2024-12-13 19:00:00'),
(135, 5, 17.65, '2024-12-14 19:00:00'),
(136, 5, 17.60, '2024-12-15 19:00:00'),
(137, 5, 17.40, '2024-12-16 19:00:00'),
(138, 5, 17.45, '2024-12-17 19:00:00'),
(139, 5, 17.40, '2024-12-18 19:00:00'),
(140, 5, 17.35, '2024-12-19 19:00:00'),
(141, 5, 17.30, '2024-12-20 19:00:00'),
(142, 5, 17.20, '2024-12-21 19:00:00'),
(143, 5, 17.10, '2024-12-22 19:00:00'),
(144, 5, 17.00, '2024-12-23 19:00:00'),
(145, 5, 17.05, '2024-12-24 19:00:00'),
(146, 5, 17.15, '2024-12-25 19:00:00'),
(147, 5, 17.20, '2024-12-26 19:00:00'),
(148, 5, 17.25, '2024-12-27 19:00:00'),
(149, 5, 17.30, '2024-12-28 19:00:00'),
(150, 5, 17.36, '2024-12-29 19:00:00'),
(151, 6, 27.55, '2024-11-30 19:00:00'),
(152, 6, 27.20, '2024-12-01 19:00:00'),
(153, 6, 27.15, '2024-12-02 19:00:00'),
(154, 6, 27.20, '2024-12-03 19:00:00'),
(155, 6, 27.25, '2024-12-04 19:00:00'),
(156, 6, 27.40, '2024-12-05 19:00:00'),
(157, 6, 27.60, '2024-12-06 19:00:00'),
(158, 6, 27.75, '2024-12-07 19:00:00'),
(159, 6, 27.40, '2024-12-08 19:00:00'),
(160, 6, 27.30, '2024-12-09 19:00:00'),
(161, 6, 27.25, '2024-12-10 19:00:00'),
(162, 6, 27.20, '2024-12-11 19:00:00'),
(163, 6, 27.15, '2024-12-12 19:00:00'),
(164, 6, 27.10, '2024-12-13 19:00:00'),
(165, 6, 26.80, '2024-12-14 19:00:00'),
(166, 6, 26.50, '2024-12-15 19:00:00'),
(167, 6, 26.20, '2024-12-16 19:00:00'),
(168, 6, 26.10, '2024-12-17 19:00:00'),
(169, 6, 26.00, '2024-12-18 19:00:00'),
(170, 6, 26.40, '2024-12-19 19:00:00'),
(171, 6, 26.50, '2024-12-20 19:00:00'),
(172, 6, 26.30, '2024-12-21 19:00:00'),
(173, 6, 26.20, '2024-12-22 19:00:00'),
(174, 6, 26.10, '2024-12-23 19:00:00'),
(175, 6, 26.05, '2024-12-24 19:00:00'),
(176, 6, 26.05, '2024-12-25 19:00:00'),
(177, 6, 26.10, '2024-12-26 19:00:00'),
(178, 6, 26.15, '2024-12-27 19:00:00'),
(179, 6, 26.15, '2024-12-28 19:00:00'),
(180, 6, 26.15, '2024-12-29 19:00:00'),
(181, 7, 120.20, '2024-12-30 19:00:00'),
(182, 7, 119.00, '2024-12-29 19:00:00'),
(183, 7, 117.60, '2024-12-28 19:00:00'),
(184, 7, 120.60, '2024-12-27 19:00:00'),
(185, 7, 121.00, '2024-12-26 19:00:00'),
(186, 7, 119.50, '2024-12-25 19:00:00'),
(187, 7, 118.20, '2024-12-24 19:00:00'),
(188, 7, 120.10, '2024-12-23 19:00:00'),
(189, 7, 119.30, '2024-12-22 19:00:00'),
(190, 7, 120.50, '2024-12-21 19:00:00'),
(191, 7, 121.20, '2024-12-20 19:00:00'),
(192, 7, 119.80, '2024-12-19 19:00:00'),
(193, 7, 118.90, '2024-12-18 19:00:00'),
(194, 7, 120.40, '2024-12-17 19:00:00'),
(195, 7, 121.50, '2024-12-16 19:00:00'),
(196, 7, 119.70, '2024-12-15 19:00:00'),
(197, 7, 120.90, '2024-12-14 19:00:00'),
(198, 7, 118.80, '2024-12-13 19:00:00'),
(199, 7, 117.90, '2024-12-12 19:00:00'),
(200, 7, 120.30, '2024-12-11 19:00:00'),
(201, 7, 119.60, '2024-12-10 19:00:00'),
(202, 7, 118.70, '2024-12-09 19:00:00'),
(203, 7, 120.00, '2024-12-08 19:00:00'),
(204, 7, 121.10, '2024-12-07 19:00:00'),
(205, 7, 120.80, '2024-12-06 19:00:00'),
(206, 7, 118.60, '2024-12-05 19:00:00'),
(207, 7, 119.40, '2024-12-04 19:00:00'),
(208, 7, 120.70, '2024-12-03 19:00:00'),
(209, 7, 121.30, '2024-12-02 19:00:00'),
(210, 7, 120.20, '2024-12-01 19:00:00');

-- --------------------------------------------------------

--
-- Table structure for table `stocks`
--

CREATE TABLE `stocks` (
  `stock_id` int(40) NOT NULL,
  `ticker_symbol` varchar(120) NOT NULL,
  `company_name` varchar(120) NOT NULL,
  `current_price` decimal(40,2) NOT NULL,
  `sector` varchar(120) NOT NULL,
  `listing_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `high_price_52w` decimal(40,2) NOT NULL,
  `low_price_52w` decimal(40,2) NOT NULL,
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `stocks`
--

INSERT INTO `stocks` (`stock_id`, `ticker_symbol`, `company_name`, `current_price`, `sector`, `listing_date`, `high_price_52w`, `low_price_52w`, `updated_at`) VALUES
(1, '2030.SR', 'SARCO', 72.00, 'Energy', '2024-12-31 10:51:14', 99.00, 68.00, '2024-12-31 10:51:14'),
(2, '2222.SR', 'SAUDI ARAMCO', 28.00, 'Energy', '2024-12-31 10:51:14', 34.00, 27.00, '2024-12-31 10:51:14'),
(3, '2380.SR', 'PETRO RABIGH', 8.00, 'Energy', '2024-12-31 10:51:14', 11.00, 7.00, '2024-12-31 10:51:14'),
(4, '2381.SR', 'ARABIAN DRILLING', 112.00, 'Energy', '2024-12-31 10:51:14', 214.00, 104.00, '2024-12-31 10:51:14'),
(5, '2382.SR', 'ADES', 17.00, 'Energy', '2024-12-31 10:51:14', 26.00, 17.00, '2024-12-31 10:51:14'),
(6, '4030.SR', 'BAHRI', 26.00, 'Energy', '2024-12-31 10:51:14', 30.00, 22.00, '2024-12-31 10:51:14'),
(7, '4200.SR', 'ALDREES', 120.00, 'Energy', '2024-12-31 10:51:14', 148.00, 100.00, '2024-12-31 10:51:14');

-- --------------------------------------------------------

--
-- Table structure for table `transactions`
--

CREATE TABLE `transactions` (
  `transaction_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `stock_id` int(40) NOT NULL,
  `transaction_type` varchar(120) NOT NULL,
  `quantity` int(11) NOT NULL,
  `price_per_share` decimal(10,2) NOT NULL,
  `total_value` decimal(10,2) NOT NULL,
  `transaction_date` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `user_info`
--

CREATE TABLE `user_info` (
  `id` int(11) NOT NULL,
  `name` varchar(40) NOT NULL,
  `Surname` varchar(40) NOT NULL,
  `DOB_Day` int(40) NOT NULL,
  `DOB_Month` varchar(40) NOT NULL,
  `DOB_Year` int(40) NOT NULL,
  `Gender` text NOT NULL,
  `Mobile_Number_Or_Email` text NOT NULL,
  `Password` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `user_info`
--

INSERT INTO `user_info` (`id`, `name`, `Surname`, `DOB_Day`, `DOB_Month`, `DOB_Year`, `Gender`, `Mobile_Number_Or_Email`, `Password`) VALUES
(1, 'Ali', 'zain', 12, 'jan', 1999, 'male', '033567', '12345'),
(2, 'Fahad', 'Mustafa', 2, 'Mar', 1962, 'Male', 'fahad@ffkwn.com', '12345');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `funds`
--
ALTER TABLE `funds`
  ADD PRIMARY KEY (`fund_id`),
  ADD KEY `fk_user_` (`user_id`);

--
-- Indexes for table `portfolio`
--
ALTER TABLE `portfolio`
  ADD PRIMARY KEY (`portfolio_id`),
  ADD KEY `fk_stock_id` (`stock_id`),
  ADD KEY `fk_user_id` (`user_id`);

--
-- Indexes for table `stock price history`
--
ALTER TABLE `stock price history`
  ADD PRIMARY KEY (`price_history_id`),
  ADD KEY `fk_stok_id` (`stock_id`);

--
-- Indexes for table `stocks`
--
ALTER TABLE `stocks`
  ADD PRIMARY KEY (`stock_id`);

--
-- Indexes for table `transactions`
--
ALTER TABLE `transactions`
  ADD PRIMARY KEY (`transaction_id`),
  ADD KEY `fk_stocks_id` (`stock_id`),
  ADD KEY `fk_users_id` (`user_id`);

--
-- Indexes for table `user_info`
--
ALTER TABLE `user_info`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `funds`
--
ALTER TABLE `funds`
  MODIFY `fund_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `portfolio`
--
ALTER TABLE `portfolio`
  MODIFY `portfolio_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;

--
-- AUTO_INCREMENT for table `stock price history`
--
ALTER TABLE `stock price history`
  MODIFY `price_history_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=211;

--
-- AUTO_INCREMENT for table `stocks`
--
ALTER TABLE `stocks`
  MODIFY `stock_id` int(40) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `transactions`
--
ALTER TABLE `transactions`
  MODIFY `transaction_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `user_info`
--
ALTER TABLE `user_info`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `funds`
--
ALTER TABLE `funds`
  ADD CONSTRAINT `fk_user_` FOREIGN KEY (`user_id`) REFERENCES `user_info` (`id`);

--
-- Constraints for table `portfolio`
--
ALTER TABLE `portfolio`
  ADD CONSTRAINT `fk_stock_id` FOREIGN KEY (`stock_id`) REFERENCES `stocks` (`stock_id`),
  ADD CONSTRAINT `fk_user_id` FOREIGN KEY (`user_id`) REFERENCES `user_info` (`id`);

--
-- Constraints for table `stock price history`
--
ALTER TABLE `stock price history`
  ADD CONSTRAINT `fk_stok_id` FOREIGN KEY (`stock_id`) REFERENCES `stocks` (`stock_id`);

--
-- Constraints for table `transactions`
--
ALTER TABLE `transactions`
  ADD CONSTRAINT `fk_stocks_id` FOREIGN KEY (`stock_id`) REFERENCES `stocks` (`stock_id`),
  ADD CONSTRAINT `fk_users_id` FOREIGN KEY (`user_id`) REFERENCES `user_info` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
